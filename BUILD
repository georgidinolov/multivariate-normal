cc_library(
	name = "multivariate-normal",
	srcs = ["MultivariateNormal.cpp"],
	hdrs = ["MultivariateNormal.hpp"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas"],
	visibility = ["//visibility:public"]
)

cc_binary(
	name = "multivariate-normal-test",
	srcs = ["multivariate-normal-test.cpp"],
	includes = ["MultivariateNormal.hpp"],
	deps = [":multivariate-normal"],
)

cc_binary(
	name = "wishart-test",
	srcs = ["wishart-test.cpp"],
	includes = ["MultivariateNormal.hpp"],
	deps = [":multivariate-normal"],
)