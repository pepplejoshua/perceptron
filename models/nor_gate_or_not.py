from .not_gate import NotGate
from .or_gate import OrGate
from typing import List


class NorGate:
    def __init__(self) -> None:
        not_weights_and_bias = "weights/not_run_1/not_gate_weights.txt"
        or_weights_and_bias = "weights/or_run_1/or_gate_weights.txt"

        not_gate = NotGate()
        not_gate.load_weights(not_weights_and_bias)
        or_gate = OrGate()
        or_gate.load_weights(or_weights_and_bias)

        self.not_gate = not_gate
        self.or_gate = or_gate

    def forward(self, x: List[int]) -> int:
        """
        Process 2 inputs using a 2-input OR gates and a 1 input NOT gate
        connected in cascade
        Args:
            x: List of 2 binary inputs [x1, x2]
        Returns:
            Binary output (0 or 1)
        """
        # OR gate processes x1 OR x2
        intermediate = self.or_gate.forward(x)

        # NOT gate processes !(x1 OR x2)
        final = self.not_gate.forward(intermediate)

        return final
