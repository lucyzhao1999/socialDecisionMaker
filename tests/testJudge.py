import sys
import os
sys.path.append(os.path.join('..', 'src'))

import unittest
from ddt import ddt, data, unpack
from resourceAllocationFairness.baseDecisionMaker import getUtilityOfEfficiency, getInequity, \
    GetUtilityOfInequity, GetBaseDecisionUtility, GetActionProbabilityFromUtility
from resourceAllocationFairness import judge as targetcode


@ddt
class testJudge(unittest.TestCase):
    def setUp(self):
        self.alphaPartial = 6
        self.alphaIA = 0.7
        self.beta = 0.003

        self.getUtilityOfInequity = GetUtilityOfInequity(getInequity)
        self.getBaseDecisionUtility = GetBaseDecisionUtility(getUtilityOfEfficiency, self.getUtilityOfInequity)
        self.getActionProbabilityFromUtility = GetActionProbabilityFromUtility(self.beta)

    @data(
        (2, [(6, -6), (-6, 6), (1, 1)]),
        (3, [(6, -6, -6), (-6, 6, -6), (-6, -6, 6), (6, 6, -6), (6, -6, 6), (-6, 6, 6), (1, 1, 1)])
    )
    @unpack
    def testAgentsWeightSet(self, numberOfAgents, trueAgentsWeightSet):
        getAgentsWeightSet = targetcode.GetAgentsWeightSet(self.alphaPartial, targetcode.getAgentsWeight)
        agentsWeightSet = getAgentsWeightSet(numberOfAgents)
        self.assertEqual(trueAgentsWeightSet, agentsWeightSet)


    @data(([(6, -6, -6), (-6, 6, -6), (-6, -6, 6), (1, 1, 1)],
           {(6, -6, -6): 1/3, (-6, 6, -6): 1/3, (-6, -6, 6): 1/3, (1, 1, 1): 0}
    ),
          ([(6, -6, -6), (1, 1, 1)],
           {(6, -6, -6): 1, (1, 1, 1): 0})
    )
    @unpack
    def testProbOfAlphaVectorGivenPartial(self, agentsWeight, trueProbOfAlphaGivenPartial):
        probOfAlphaGivenPartial = targetcode.getProbOfAlphaVectorGivenPartial(agentsWeight)
        self.assertEqual(trueProbOfAlphaGivenPartial, probOfAlphaGivenPartial)


    @data(([(6, -6, -6), (-6, 6, -6), (-6, -6, 6), (1, 1, 1)],
           {(6, -6, -6): 0, (-6, 6, -6): 0, (-6, -6, 6): 0, (1, 1, 1): 1}
    ),
          ([(6, -6, -6), (1, 1, 1)],
           {(6, -6, -6): 0, (1, 1, 1): 1})
    )
    @unpack
    def testProbOfAlphaVectorGivenImpartial(self, agentsWeight, trueProbOfAlphaGivenImpartial):
        probOfAlphaGivenImpartial = targetcode.getProbOfAlphaVectorGivenImpartial(agentsWeight)
        self.assertEqual(trueProbOfAlphaGivenImpartial, probOfAlphaGivenImpartial)



    def tearDown(self):
        pass

if __name__ == "__main__":
    judgeTest = unittest.TestLoader().loadTestsFromTestCase(testJudge)
    unittest.TextTestRunner(verbosity=2).run(judgeTest)
