from sklearn import tree
# features as the input
features = [[140, 0], [130, 0], [150, 1], [170,1]]

# labels as the output
labels = [0, 0 , 1 , 1]

# clf = classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# prints must have params
print (clf.predict([[150,0]]))
# wtf!!!!!!!
# Post thighs!!!!