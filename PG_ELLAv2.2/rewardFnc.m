function rew = rewardFnc(x, u)

rew=-sqrt(x'*x)-sqrt(u'*u);