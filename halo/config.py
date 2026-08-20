from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


DT_THRESHOLDS_DEFAULT: Tuple[float, ...] = (0.01, 0.1, 1.0)


@dataclass(frozen=True)
class FeatureConfig:
    use_dir: bool = True
    use_dt_bucket: bool = True
    use_dir_sign_for_len: bool | None = None
    dt_thresholds: Sequence[float] = DT_THRESHOLDS_DEFAULT
    max_pkt_len: float = 1500.0

    def __post_init__(self) -> None:
        if self.use_dir_sign_for_len is None:
            object.__setattr__(self, "use_dir_sign_for_len", self.use_dir)
        object.__setattr__(self, "dt_thresholds", tuple(float(x) for x in self.dt_thresholds))
        object.__setattr__(self, "max_pkt_len", float(self.max_pkt_len))

    @property
    def num_dt_buckets(self) -> int:
        return len(self.dt_thresholds) + 1 if self.use_dt_bucket else 1

    @property
    def num_dirs(self) -> int:
        return 2 if self.use_dir else 1

    @property
    def num_states(self) -> int:
        return self.num_dt_buckets * self.num_dirs

    @property
    def state_pad_id(self) -> int:
        return self.num_states

    @classmethod
    def from_args(cls, args, max_pkt_len: float | None = None) -> "FeatureConfig":
        return cls(
            use_dir=not bool(getattr(args, "no_dir", False)),
            use_dt_bucket=not bool(getattr(args, "no_dt_bucket", False)),
            max_pkt_len=float(max_pkt_len if max_pkt_len is not None else getattr(args, "max_pkt_len", 1500.0)),
        )
