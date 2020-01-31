# Information About ndarray

### Shape

ndarray의 모양임

Shape (a, )은  (a, 1)과 동일한 것임 생략이라 볼수 있음

ex)

test_np= np.array([1, 2, 3, 4, 5])

shape = test_np.shape

출력: (5,)

---

### dtype

자료형 같은 개념인데 효율적인 메모리 관리를 위하여 하는 것임

수동으로 하는것이 최적화에 좋음

[https://docs.scipy.org/doc/numpy/user/basics.types.html](https://docs.scipy.org/doc/numpy/user/basics.types.html) 에 numpy data type 정리해놓음

---

### size

ndarray의 용량을 파악하는 것임

A X B X C 차원이라면 size는 A*B*C가 된다

### itemsize

각각의 요소가 어떤 data type이며 그 data가 메모리를 얼마나 차지하는지 알려준다

그래서 size * itemsize를 하게 되면 전체 차지하는 메모리가 나오게 된다