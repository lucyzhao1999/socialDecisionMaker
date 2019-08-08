import numpy as np

class GetReputationBoostParam:
    def __init__(self, distributionParam, boostMultiplier):
        self.distributionParam = distributionParam
        self.boostMultiplier = boostMultiplier

    def __call__(self, probOfPublic):
        reputationBoostParam = self.boostMultiplier * np.power(probOfPublic, self.distributionParam)

        return reputationBoostParam


class GetReputationHurtParam:
    def __init__(self, distributionParam, hurtMultiplier):
        self.distributionParam = distributionParam
        self.hurtMultiplier = hurtMultiplier

    def __call__(self, probOfPublic):
        reputationHurtParam = self.hurtMultiplier* np.power(probOfPublic, self.distributionParam)

        return reputationHurtParam