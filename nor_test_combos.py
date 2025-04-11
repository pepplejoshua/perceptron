import numpy as np
from models.nor_gate import NorGate


class NorBasedGates:
    def __init__(self):
        """Initialize with trained NOR gate"""
        self.nor = NorGate()
        self.nor.load_weights("weights/nor_run_1/nor_gate_weights.txt")

    def NOT(self, x: int) -> int:
        """NOT(x) = NOR(x,x)"""
        return self.nor.forward([x, x])

    def OR(self, x: int, y: int) -> int:
        """OR(x,y) = NOT(NOR(x,y)) = NOR(NOR(x,y), NOR(x,y))"""
        nor_result = self.nor.forward([x, y])
        return self.NOT(nor_result)  # NOT of NOR gives OR

    def AND(self, x: int, y: int) -> int:
        """AND(x,y) = NOR(NOR(x,x), NOR(y,y))
        Using De Morgan's law: x AND y = NOT(NOT(x) OR NOT(y))"""
        not_x = self.NOT(x)
        not_y = self.NOT(y)
        return self.nor.forward([not_x, not_y])

    def NAND(self, x: int, y: int) -> int:
        """NAND(x,y) = NOT(AND(x,y))"""
        and_result = self.AND(x, y)
        return self.NOT(and_result)

    def XOR(self, x: int, y: int) -> int:
        """XOR(x,y) = AND(OR(x,y), NAND(x,y))"""
        or_result = self.OR(x, y)
        nand_result = self.NAND(x, y)
        return self.AND(or_result, nand_result)


def test_gate(gate_func, name: str):
    """Test a 2-input logic gate with all possible inputs"""
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    print(f"\nTesting {name} gate (built from NOR gates):")
    print("x1 | x2 | Output")
    print("-" * 20)

    for x in test_inputs:
        output = gate_func(x[0], x[1])
        print(f" {x[0]}  | {x[1]}  |   {output}")


def test_not_gate(not_func):
    """Test NOT gate with both possible inputs"""
    test_inputs = np.array([0, 1])

    print("\nTesting NOT gate (built from NOR gate):")
    print("Input | Output")
    print("-" * 15)

    for x in test_inputs:
        output = not_func(x)
        print(f"  {x}   |   {output}")


if __name__ == "__main__":
    gates = NorBasedGates()

    # Test NOT gate
    test_not_gate(gates.NOT)

    # Test all 2-input gates
    test_gate(gates.OR, "OR")
    test_gate(gates.AND, "AND")
    test_gate(gates.NAND, "NAND")
    test_gate(gates.XOR, "XOR")

    print("\nAll gates above were built using only NOR gates!")
    print("Truth tables should match:")
    print("NOT:  1,0")
    print("OR:   0,1,1,1")
    print("AND:  0,0,0,1")
    print("NAND: 1,1,1,0")
    print("XOR:  0,1,1,0")
