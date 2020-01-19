# 200119

### ndarray

- numpy에서 사용하는 data structure

---

### ndarray를 만드는 방법

1. np.array(shape,dtype=float,buffer=None,offset=0,strides=None,order=None)
    - python list를 이용해서 ndarray 만듬
    - ndarray를 넣을수도 있지만 list 주로 사용

2. np.zeros(shape,dtype=float,order='C')
    - shape을 입력받고 shape에 요소들은 0으로 채운다
    - 입력받는 shape은 차원에 상관없이 가능하다

3. np.ones(shape = shape,dtype = None, order = 'C')
    - np.zeros와 다르게 0이 아닌 1로 채움

4. np.full(shape, fill_value, dtype = None, order = 'C')
    - 0 과 1이 아닌 다른 특정값으로 초기화한다
    - parameter의  fill_value로 초기화한다

5. np.empty(shape, dtype = float, order = 'C')
    - 특정값으로 초기화 하는것이 아니라 shape에 따라 형태만 잡음
    - memory만 잡기 떄문에 처음엔 쓰레기값으로 채워짐

---

### 기존 ndarray와 동일한 shape을 가지는 ndarray 생성 방법

1. np.ones_like(a,dtype=None,order='K',subok=True)
    - input과 동일한 shape을 가지고 요소들은 1로 초기화

2. np.zeros_like(a,dtype=None,order='K',subok = True, shape = None)
    - input과 동일한 shape을 가지고 요소들은 0으로 초기화

3. np.full_like(a,fill_value,dtype=None,order='K',subok=True,shape=None)
    - input과 동일한 shape에 요소들은 입력받은 fill_value로 초기화 한다

4. np.empty_like(proto_type,dtype=None,order='K',subok=True,shape=None)
    - input과 동일한 shape을 만들고 값은 초기화하지 않는다
    - 값을 초기화하지 않아서 쓰레기 값이 담기거나 ndarray의 memory에 전에 있던 값이 담김