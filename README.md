# sac-walker

This is an implementation of Soft Actor critic using JAX as the main framework for the neural nets.


These are the main highlights of this implementation:
- Implementing neural nets using `flax.linen`
- Jitting class methods to speed up functions and so that `grad` operator can be used, rather than using pure functions
- Using the `optax` adam optimizer to perform gradient descent
- Polyak averaging to update weights on the target networks
- Builder pattern to construct all the components for the training algorithm
