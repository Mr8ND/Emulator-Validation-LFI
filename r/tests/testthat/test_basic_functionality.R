library(testthat)
library(Regr2STest)

test_that("Test does not fail for the same sample", {
  first_sample <- rnorm(500)
  second_sample <- first_sample
  
  model <- Regr2STest::TwoSampleRegressionTest$new(classifier_name='Log. Regr.')
  p_val <- model$ts_test(first_sample, second_sample)
  expect_true(p_val > 0.01)
})

test_that("Test does not fail for the sample widely different", {
  first_sample <- rnorm(500)
  second_sample <- runif(500, -5, -4)
  
  model <- TwoSampleRegressionTest$new(classifier_name='Log. Regr.')
  p_val <- model$ts_test(first_sample, second_sample)
  expect_true(p_val <= 0.05)
})