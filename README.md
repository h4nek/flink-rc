# flink-rc
A reservoir computing library for Apache Flink framework.

Developed as part of a bachelor thesis project (link to be added).
Developed in Java, using Maven structure

IMPORTANT: For the test examples to work correctly, they need to be downloaded and put into the appropriate paths (didn't include them to avoid possible copyright infringements). For now, their descriptions (together with the reference to download from) can be found in the *test/java/lm* folder versions.

The project is not very clean at the moment..

### Typical library usage:
1.  Create a `DataStream` (or `DataSet`) of `<Tuple2<Long, List<Double>>`
    - -> Represents the sequence of input vectors **u**(t)
2.  Call `.map(ESNReservoirSparse)`  on the `DataSet/DataStream` (collection)
    - -> This gives us sequence of state vectors **x**(t) (more precisely, typically the concatenation [1 **u**(t) **x**(t)])
3.  Split the collection on the training and testing part (if we didn’t do it earlier)
4.  Using the training collection, call `LinearRegression(Primitive).fit(args)`
    - -> Obtain the vector of optimal regression coefficients 𝛂∗
5.  Using the testing collection and 𝛂∗, call `LinearRegression(Primitive).predict(args)`
    - -> This gives us the final scalar result (prediction) 𝑦 ̂(t)
6.  (Optional) Compare our predictions 𝑦 ̂(t) with the real values y(t) (when/if they are available)
