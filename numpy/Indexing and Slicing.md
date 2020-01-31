# Indexing and Slicing

기존의 파이썬 리스트의 경우 

인덱스를 뽑아내거나 인덱스의 범위를 통해서 slicing이 가능했음

ndarray도 동일하게 적용된다

---

하지만 리스트의 경우 2차원 이상에서 행렬처럼 접근을 할수가 없다

예를들어

matrix = [[1,2], [3,4,], [5,6], [7,8]]

print(matrix[1,1])

output: error

matrix[1][1] 이런식으로 접근해야된다

하지만 ndarray는 가능하다