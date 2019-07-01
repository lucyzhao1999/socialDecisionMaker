import unittest
from ddt import ddt, data, unpack


@ddt
class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.alphaPartial = 6
        self.alphaIA = 0.7

    @data(
        (3, 1, [(6, -6, -6), (-6, 6, -6), (-6, -6, 6), (1,1,1)]),
        (3, 2, [(6, 6, -6), (6, -6, 6), (-6, 6, 6), (1,1,1)]),
    )
    @unpack
    def testAgentsWeightSet(self, numberOfAgents, partialAgentNumber, trueAgentsWeightSet):
        agentsWeightSet = getAgentsWeightSet(self.alphaPartial, numberOfAgents, partialAgentNumber)
        self.assertEqual(trueAgentsWeightSet, agentsWeightSet)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
