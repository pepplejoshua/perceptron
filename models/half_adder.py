#            |    Sum    | Carry |
# Half Adder | XOR(x, y) | AND() |
from .and_gate import AndGate
from .xor_gate_or_nand_and import XorGate
from typing import List, Tuple


class HalfAdder:
    def __init__(self) -> None:
        xor = XorGate()
        _and = AndGate()
        _and.load_weights("weights/and_run_1/and_gate_weights.txt")
        self.xor = xor
        self.and_ = _and

    def forward(self, x: List[int]) -> Tuple[int, int]:
        """
        Process 2 inputs using a 2-input XOR gate, and a 2-input
        AND gate since HA = [sum = XOR(x, y), carry = NAND(x, y)]
        Args:
            x: List of 2 binary inputs [x1, x2]
        Returns:
            Binary output (0 or 1)
        """
        sum = self.xor.forward(x)
        carry = self.and_.forward(x)
        return sum, carry
