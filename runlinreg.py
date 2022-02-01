
import pandas as pd 
from sklearn import linear_model

data = pd.read_csv("datatom.csv")

#print(data.describe())
#.values converts panda objects to matrix, since sklearn doesn't like panda
target = data.iloc[:,0].values

#print(target)

indvar = data.iloc[:,3:9].values

#print(indvar)

machine = linear_model.LinearRegression()
print(machine)
machine.fit(indvar,target)
print(machine)

new_data = [
	[-0.5,1.1,0.88,0.4,3,0],
	[0.6,1.4,-0.1,1,2,1]
]

new_target = machine.predict(new_data)
print(new_target)