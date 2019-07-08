import sys
import os
sys.path.append(os.path.join('..', 'src'))

import baseDecisionMaker as targetcode
import unittest
from ddt import ddt, data, unpack

@ddt
class TestSocialDecisionMaker(unittest.TestCase):

    def setUp(self):
        self.lambdaInput = 0.7
        self.alphaAIList = [0.7]
        self.absoluteEffort = 1
        self.beta = 0.003
        self.alphaPartial = 6
        self.alphaPA = 1350
        self.judgePrior = 0.5

        self.alphaIA = 0.7
        self.createReward = targetcode.CreateReward(targetcode.createPartialAllocationList)

        self.getUtilityOfInequity = targetcode.GetUtilityOfInequity(targetcode.getInequity)
        self.getBaseDecisionUtility = targetcode.GetBaseDecisionUtility(targetcode.getUtilityOfEfficiency, self.getUtilityOfInequity)

        self.getActionProbabilityGivenAlpha = targetcode.GetActionProbabilityFromUtility(self.beta)



    @data(([1,2], 5, 200, 800, [200, 800, 800, 200, 200]),
          ([1], 3, 300, 500, [300, 500, 300]),
          ([0,1,2], 4, 300, 500, [500, 500, 500, 300])
          )
    @unpack
    def testPartialAllocation(self, highBonusAgentIndexList, numberOfAgents, lowBonus, highBonus, groundtruth_allocationList):
        allocationList = targetcode.createPartialAllocationList(highBonusAgentIndexList, numberOfAgents, lowBonus, highBonus)
        self.assertEqual(allocationList, groundtruth_allocationList)

    @data((800, 200, 100, 1, 3,
           {'Action1': [800, 200, 200, 200],
             'Action2': [200, 800, 200, 200],
             'Action3': [200, 200, 800, 200],
             'Action4': [200, 200, 200, 800],
             'ActionEqual': [100, 100, 100, 100]}
           ),
          (800, 200, 100, 2, 2,
           {'Action1': [800, 800, 200, 200],
             'Action2': [800, 200, 800, 200],
             'Action3': [800, 200, 200, 800],
             'Action4': [200, 800, 800, 200],
             'Action5': [200, 800, 200, 800],
             'Action6': [200, 200, 800, 800],
             'ActionEqual': [100, 100, 100, 100]}
            )
          )
    @unpack
    def testReward(self, highBonus, lowBonus, equalBonusList, highBonusAgentCount, lowBonusAgentCount, trueReward):
        reward = self.createReward(highBonus, lowBonus, equalBonusList, highBonusAgentCount, lowBonusAgentCount)
        self.assertEqual(reward, trueReward)

    @data(({'Action1': [800, 200, 200, 200],
            'Action2': [200, 800, 200, 200],
            'Action3': [200, 200, 800, 200],
            'Action4': [200, 200, 200, 800],
            'Action5': [100, 100, 100, 100]},
           [1,1,1,1],
           {'Action1': 1400,
            'Action2': 1400,
            'Action3': 1400,
            'Action4': 1400,
            'Action5': 400}
    ),
          ({'Action1': [800, 200, 200, 200],
            'Action2': [200, 800, 200, 200],
            'Action3': [200, 200, 800, 200],
            'Action4': [200, 200, 200, 800],
            'Action5': [100, 100, 100, 100]},
           [-1, 1, 0, 1],
           {'Action1': -400,
            'Action2': 800,
            'Action3': 200,
            'Action4': 800,
            'Action5': 100}
           )
    )
    @unpack
    def testUtilityOfEfficiency(self, reward, agentWeights, trueUtilityOfEfficiency):
        utilityOfEfficiency = targetcode.getUtilityOfEfficiency(reward, agentWeights)
        self.assertEqual(utilityOfEfficiency, trueUtilityOfEfficiency)


    @data(([3,4], [200, 100],500 ),
          ([3,4,5], [200, 100, 100], 1300),
          ([-1, 1, 1], [300, 200, 100], 1000)
    )
    @unpack
    def testInequity(self, merit, reward, trueInequity):
        inequity = targetcode.getInequity(merit, reward)
        self.assertEqual(inequity, trueInequity)

    @data(({'Action1': [800, 200, 200],
            'Action2': [200, 800, 200],
            'Action3': [100, 100, 100]},
           [1,1,1],
           0.5,
           {'Action1': 600,
            'Action2': 600,
            'Action3': 0}
          ))
    @unpack
    def testInequity(self, reward, merit, alphaIA, trueUtilityOfInequity):
        getUtilityOfInequity = targetcode.GetUtilityOfInequity(targetcode.getInequity)
        utilityOfInequity = getUtilityOfInequity(reward, merit, alphaIA)
        self.assertEqual(trueUtilityOfInequity, utilityOfInequity)


    @data(({'Action1': [800, 200, 200],
            'Action2': [200, 800, 200],
            'Action3': [100, 100, 100]},
          [[2,-2,2],0.5],
          [1,1,1],
          {'Action1': 1000,
           'Action2': -1400,
           'Action3': 200}
          ),(
            {'Action1': [200, 100, 100],'Action2': [100, 200, 100],'Action3': [100, 100, 200]},
            [[1, -1, 1], 0.5],
            [1, 1, 1],
            {'Action1': 100, 'Action2': -100, 'Action3': 100}
    )
          )
    @unpack
    def testBaseUtility(self, reward, alphaVector, merit, trueBaseUtility):
        baseUtility = self.getBaseDecisionUtility(reward, alphaVector, merit)
        self.assertEqual(trueBaseUtility, baseUtility)


    def testActionProb(self):
        reward = {'Action1': [800, 200, 200],
                  'Action2': [200, 800, 200],
                  'Action3': [100, 100, 100]}
        alphaVector = [[2,-2,2],0.5]
        merit = [1,1,1]
        baseUtility = self.getBaseDecisionUtility(reward, alphaVector, merit)
        actionProb = self.getActionProbabilityGivenAlpha(baseUtility)
        self.assertEqual(sum(actionProb.values()), 1)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()

