from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vocab:
    """
    Vocabulary for the toy group-prediction task.

    Tokens:
        0                              -> BOS
        1 .. N                         -> cat_0 .. cat_{N-1}
        N+1 .. N + N*k                 -> cat_x_grp_y  (index = 1 + N + x*k + y)

    Training sequences have the form:
        [BOS, cat_x, cat_x_grp_y, cat_z]  ->  cat_z_grp_y

    The model must learn: whatever group index y is shown in the in-context
    demonstration (cat_x_grp_y), predict that same group index for cat_z.
    """

    n_categories: int
    k: int  # number of group slots per category

    @property
    def vocab_size(self) -> int:
        return 1 + self.n_categories + self.n_categories * self.k

    def bos(self) -> int:
        return 0

    def cat(self, x: int) -> int:
        return 1 + x

    def grp(self, x: int, y: int) -> int:
        return 1 + self.n_categories + x * self.k + y

    def decode_cat(self, token_id: int) -> int | None:
        """Returns x if token_id is cat_x, else None."""
        if 1 <= token_id <= self.n_categories:
            return token_id - 1
        return None

    def decode_grp(self, token_id: int) -> tuple[int, int] | None:
        """Returns (x, y) if token_id is cat_x_grp_y, else None."""
        start = 1 + self.n_categories
        end = start + self.n_categories * self.k
        if start <= token_id < end:
            offset = token_id - start
            return offset // self.k, offset % self.k
        return None

    def token_name(self, token_id: int) -> str:
        if token_id == 0:
            return "BOS"
        cat = self.decode_cat(token_id)
        if cat is not None:
            return f"cat_{cat}"
        grp = self.decode_grp(token_id)
        if grp is not None:
            x, y = grp
            return f"cat_{x}_grp_{y}"
        return f"tok_{token_id}"
