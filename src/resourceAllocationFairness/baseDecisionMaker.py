import math
from itertools import combinations


def createPartialAllocationList(highBonusAgentIndexList, numberOfAgents, lowBonus, highBonus):
    allocationValueList = [lowBonus] * numberOfAgents
    for highBonusAgentIndex in highBonusAgentIndexList:
        allocationValueList[highBonusAgentIndex] = highBonus
    return allocationValueList


class CreateReward:
    def __init__(self, createPartialAllocationList):
        self.createPartialAllocationList = createPartialAllocationList

    def __call__(self, highBonus, lowBonus, equalBonus, highBonusAgentCount, lowBonusAgentCount):
        numberOfAgents = highBonusAgentCount + lowBonusAgentCount
        highBonusAgentIndexList = list(combinations(list(range(numberOfAgents)), highBonusAgentCount))
        fullAllocationList = [self.createPartialAllocationList(highBonusAgentList, numberOfAgents, lowBonus, highBonus) for highBonusAgentList in highBonusAgentIndexList]
        fullAllocationList.append([equalBonus] * numberOfAgents)

        actionNames = ["Action" + str(index + 1) for index in range(len(fullAllocationList) - 1)]
        actionNames.append("ActionEqual")
        reward = dict(zip(actionNames, fullAllocationList))

        return reward


def getUtilityOfEfficiency(reward, agentWeights):
    utilityOfEfficiencyDict = {action: sum([reward * weight for reward, weight in zip(reward[action], agentWeights)]) for action in reward.keys()}
    return utilityOfEfficiencyDict


def getInequity(merit, reward):
    inequitySum = 0
    agentCount = len(merit)
    for i in range(agentCount):
        for j in range(i + 1, agentCount):
            inequitySum = inequitySum + abs(reward[i] * merit[j] - reward[j] * merit[i])
    return inequitySum


class GetUtilityOfInequity:
    def __init__(self, getInequity):
        self.getInequity = getInequity

    def __call__(self, reward, merit, alphaIA):
        actionList = list(reward.keys())
        totalInequity = [self.getInequity(merit, reward[action]) for action in actionList]
        weightedTotalInequity = {action: inequity * alphaIA for action, inequity in zip(actionList, totalInequity)}
        return weightedTotalInequity


class GetBaseDecisionUtility:
    def __init__(self, getUtilityOfEfficiency, getUtilityOfInequity):
        self.getUtilityOfEfficiency = getUtilityOfEfficiency
        self.getUtilityOfInequity = getUtilityOfInequity

    def __call__(self, reward, alphaVector, merit):
        agentWeights, alphaIA = alphaVector
        utilityOfEfficiency = self.getUtilityOfEfficiency(reward, agentWeights)
        utilityOfInequity = self.getUtilityOfInequity(reward, merit, alphaIA)
        baseDecisionUtility = {action: utilityOfEfficiency[action] - utilityOfInequity[action] for action in utilityOfEfficiency.keys()}
        return baseDecisionUtility


class GetActionProbabilityFromUtility:
    def __init__(self, beta):
        self.beta = beta
    def __call__(self, baseUtility):
        weightedUtility = {action: math.exp(baseUtility[action] * self.beta) for action in baseUtility.keys()} # e to the power of baseutility
        sumOfWeightedUtility = sum([weightedUtility[action] for action in baseUtility.keys()])
        normalizedActionProb = {action: weightedUtility[action] / sumOfWeightedUtility for action in baseUtility.keys()}
        return normalizedActionProb










