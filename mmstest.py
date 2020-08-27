from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
a = [[1],[20],[32],[4]]
b = [[1],[1],[4],[4]]

mms.fit(a)

a = mms.transform(a)
b = mms.transform(b)

print(a)
print(b)