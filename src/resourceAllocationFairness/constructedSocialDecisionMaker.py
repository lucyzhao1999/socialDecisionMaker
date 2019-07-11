class GetSingleConstructedActionProb:
    def __init__(self, alphaPA, getActionProbabilityFromUtility):
        self.alphaPA = alphaPA
        self.getActionProbabilityFromUtility = getActionProbabilityFromUtility

    def __call__(self, baseUtility, probOfPartialGivenAction):
        actionList = baseUtility.keys()

        constructedUtility = {action: baseUtility[action] - self.alphaPA * probOfPartialGivenAction[action]
                              for action in actionList}

        constructedActionProb = self.getActionProbabilityFromUtility(constructedUtility)
        return constructedActionProb

