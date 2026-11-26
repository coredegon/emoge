"""`FuzzyColors`: a color model whose cores learn, as a layer of a network."""

from __future__ import annotations

import itertools

import torch

from emoge.color.model import ColorModel
from emoge.fuzzy.math.norms import product
from emoge.fuzzy.partitions import Partition, partition

# The share of a channel a point core or an end gap starts with: softmax never gives 0.
_TINY = 1e-6


class FuzzyColors(torch.nn.Module):
    """A color model whose cores learn: the channels of its space to its colors.

    ```python
    layer = FuzzyColors(COLIBRI)   # starts as COLIBRI
    layer(values)                  # (..., channels) -> (..., len(colors))
    layer.to_model()               # a ColorModel with the cores learned
    ```

    `values` are the channels of the model's space, `model.space.from_rgb` of
    the pixels, last. The variables are partitions joined with the product, as
    in COLIBRI. A partition learns the lengths of its cores and of the gaps
    between them, so its cores keep their order and its degrees sum to 1; one
    partition shared by several families, through `select`, learns once.
    """

    def __init__(self, model: ColorModel) -> None:
        super().__init__()
        if model.and_norm is not product:
            raise ValueError("a FuzzyColors model joins its words with the product")
        self.name, self.space, self.colors = model.name, model.space, model.colors
        partitions: list[Partition] = []
        # For every family: the partition and the terms of each of its variables.
        self._families: list[tuple[str, list[tuple[int, list[int]]]]] = []
        for family, variables in model.families.items():
            words = []
            for variable in variables:
                if not isinstance(variable, Partition):
                    raise TypeError(f"{variable} is not made by partition")
                index = next(
                    (i for i, p in enumerate(partitions) if _same(p, variable)), None
                )
                if index is None:
                    index = len(partitions)
                    partitions.append(variable)
                names = list(variable.cores)
                words.append((index, [names.index(term) for term in variable.terms]))
            self._families.append((family, words))
        self._learned = [_Learned(p) for p in partitions]
        self.partitions = torch.nn.ModuleList(self._learned)  # holds their parameters
        self._channels = [self.space.channels.index(p.channel) for p in partitions]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return the degree of every color: `(..., len(colors))`."""
        words = [
            learned(values[..., channel])
            for learned, channel in zip(self._learned, self._channels, strict=True)
        ]
        blocks = []
        for _, variables in self._families:
            degrees = torch.ones_like(values[..., :1])
            for index, terms in variables:
                # Every color so far joined with every term of this variable.
                joined = degrees[..., :, None] * words[index][..., terms][..., None, :]
                degrees = joined.flatten(-2)
            blocks.append(degrees)
        return torch.cat(blocks, dim=-1)

    def drift(self) -> torch.Tensor:
        """Return how far the cores have moved from where they started.

        The mean square of the moves of their ends, each a share of its
        channel. Added to a loss, `loss + weight * layer.drift()`, it keeps the
        words near their meaning, the nearer the larger the weight.
        """
        return torch.stack([learned.drift() for learned in self._learned]).mean()

    def to_model(self) -> ColorModel:
        """Return a `ColorModel` with the cores as learned now."""
        made = [learned.partition() for learned in self._learned]
        families = {
            family: [
                made[index].select(*(list(made[index].cores)[t] for t in terms))
                for index, terms in variables
            ]
            for family, variables in self._families
        }
        return ColorModel(self.name, self.space, families)


def _same(a: Partition, b: Partition) -> bool:
    return a.channel == b.channel and a.cores == b.cores


class _Learned(torch.nn.Module):
    """A partition whose cores learn.

    It learns the lengths of its cores and of the gaps between them and, on a
    circle, where its first core starts.
    """

    initial: torch.Tensor  # where the ends of the cores started

    def __init__(self, made: Partition) -> None:
        super().__init__()
        self.channel, self.names = made.channel, list(made.cores)
        low, high = self.channel.low, self.channel.high
        self.span = high - low
        bounds = [bound for core in made.cores.values() for bound in core]
        if self.channel.periodic:
            # From the first core round: core, gap, core, gap, ... back to it.
            around = itertools.pairwise([*bounds, bounds[0]])
            lengths = [(b - a) % self.span for a, b in around]
            self.start = torch.nn.Parameter(torch.tensor(float(bounds[0])))
        else:
            # Gap from low, core, gap, core, ... gap to high.
            points = [low, *bounds, high]
            lengths = [b - a for a, b in itertools.pairwise(points)]
        start_lengths = torch.tensor(lengths, dtype=torch.float32) + _TINY * self.span
        self.lengths = torch.nn.Parameter(torch.log(start_lengths))
        self.register_buffer("initial", self._bounds().detach().clone())

    def drift(self) -> torch.Tensor:
        """Return the mean square move of the ends of the cores, as shares."""
        moved = self._bounds() - self.initial
        if self.channel.periodic:  # the shorter way round
            moved = torch.remainder(moved + self.span / 2, self.span) - self.span / 2
        return (moved / self.span).square().mean()

    def _bounds(self) -> torch.Tensor:
        """Where every core starts and ends, in order: `(terms, 2)`."""
        ends = torch.cumsum(torch.softmax(self.lengths, dim=0) * self.span, dim=0)
        if self.channel.periodic:
            return (self.start + torch.cat([ends.new_zeros(1), ends[:-1]])).view(-1, 2)
        return (self.channel.low + ends[:-1]).view(-1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the degree of `x` in every term: `(..., terms)`."""
        bounds = self._bounds()
        starts, ends = bounds[:, 0], bounds[:, 1]
        if not self.channel.periodic:
            # A term is the rise into it less the rise into the next one.
            rise = ((x[..., None] - ends[:-1]) / (starts[1:] - ends[:-1])).clamp(0, 1)
            ones = torch.ones_like(rise[..., :1])
            steps = torch.cat([ones, rise, torch.zeros_like(ones)], dim=-1)
            return steps[..., :-1] - steps[..., 1:]
        # Around the circle, measured from where the rise into each term starts.
        rise_from = torch.roll(ends, 1)
        rise = torch.remainder(starts - rise_from, self.span)
        fall = torch.remainder(torch.roll(starts, -1) - ends, self.span)
        along = torch.remainder(x[..., None] - rise_from, self.span)
        rising = (along / rise).clamp(0, 1)
        falling = ((rise + ends - starts + fall - along) / fall).clamp(0, 1)
        return torch.minimum(rising, falling)

    def partition(self) -> Partition:
        """Return the partition with the cores as learned now."""
        low = self.channel.low
        bounds = self._bounds().detach().double().tolist()
        if self.channel.periodic:
            bounds = [[low + (b - low) % self.span for b in core] for core in bounds]
        cores = {name: (a, b) for name, (a, b) in zip(self.names, bounds, strict=True)}
        return partition(self.channel, cores)
