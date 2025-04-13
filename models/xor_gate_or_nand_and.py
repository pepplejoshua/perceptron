# XOR = AND(OR(x, y), NAND(x, y))
from .and_gate import AndGate
from .or_gate import OrGate
from .nand_gate import NandGate
from typing import List


class XorGate:
    def __init__(self) -> None:
        and_ = AndGate()
        nand = NandGate()
        or_ = OrGate()

        and_.load_weights("weights/and_run_1/and_gate_weights.txt")
        nand.load_weights("weights/nand_run_1/nand_gate_weights.txt")
        or_.load_weights("weights/or_run_1/or_gate_weights.txt")

        self._and = and_
        self.nand = nand
        self._or = or_

    def forward(self, x: List[int]) -> int:
        """
        Process 2 inputs using a 2-input NAND gate, a 2-input OR gate, and
        a 2-input NAND gate since XOR = AND(OR(x, y), NAND(x, y))
        Args:
            x: List of 2 binary inputs [x1, x2]
        Returns:
            Binary output (0 or 1)
        """
        or_interm = self._or.forward(x)
        nand_interm = self.nand.forward(x)
        final = self._and.forward([or_interm, nand_interm])
        return final
