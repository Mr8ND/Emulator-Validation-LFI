library(randomForest)

#' TwoSampleRegressionTest
#'
#' The TwoSampleRegressionTest class. Provides functions for computing the
#' two sample regression test
#'
#'
#' @section Usage:
#' \preformatted{model <- NNKCDE$new(classifier_name)
#' model$ts_test(first_sample, second_sample, n_bootstrap)
#' }
#'
#' @section Arguments:
#' \code{classifier_name} Name of the classifier used in the regression test.
#' The availbility of classifiers is dependent on what it has been implemented.
#'
#' \code{first_sample} First sample of the two sample testing test
#' 
#' \code{second_sample} Second sample of the two sample testing test
#' 
#' \code{n_bootstrap} Number of boostrap repetitions for calculating the p-value
#' of the two sample test.
#'
#' @section Methods:
#' \code{$new(classifier_name)} Initializes a new TwoSampleRegressionTest
#'  object.
#'
#' \code{$regstat(X, y} function to calculate the test statistic used in the
#' test. X is the matrix of covariates, while y is a boolean vector indicating
#' which sample the datum belongs to (either first or second sample)
#' 
#' \code{$ts_test(first_sample, second_sample, n_bootstrap)} Calculates the 
#' p-value for the two sample regression test.
#'
#' @name TwoSampleRegressionTest
#' @examples
#' \dontrun{
#' model <- TwoSampleRegressionTest$new('Log Regr.')
#' model$ts_test(rnorm(500), rnorm(500))
#' }
#' @export
#' @importFrom R6 R6Class
TwoSampleRegressionTest <- R6::R6Class("TwoSampleRegressionTest", #nolint
                                        public = list(
                                          classifier_name = NULL))

TwoSampleRegressionTest$set("public", "initialize",
                             function(classifier_name) {
                               class_vec <- c('Log. Regr.', 'Random Forest')
                               if (classifier_name %in% class_vec == FALSE){
                                 error_message <- paste('Only ', class_vec[1],
                                                        ',', class_vec[2],
                                                        'are available.')
                                 stop(error_message)
                               }
                               self$classifier_name <- classifier_name
                             })

TwoSampleRegressionTest$set("public", "regstat",
                            function(X, y) {

                              if (self$classifier_name == 'Log. Regr.'){
                                mod <- stats::glm(y ~ X, family = "binomial")
                                prob <- as.numeric(
                                  predict(mod, type = "response"))
                              } else if (
                                self$classifier_name == 'Random Forest'){
                                mod <- randomForest::randomForest(
                                  X, as.factor(y))
                                prob <- predict(mod, type = "prob")[, 2]
                              }
                              return(mean((prob - mean(y)) ^ 2))
                            })

TwoSampleRegressionTest$set("public", "ts_test",
                           function(first_sample, second_sample,
                                    n_bootstrap = 100) {
                             first_sample <- as.matrix(first_sample)
                             second_sample <- as.matrix(second_sample)
                             
                             y <- c(rep(0, nrow(first_sample)), 
                                    rep(1, nrow(second_sample)))
                             X <- rbind(first_sample, second_sample)
                             
                             stat <- self$regstat(X, y)
                             
                             null <- replicate(n_bootstrap, {
                               y <- sample(y)
                               return(self$regstat(X, y))
                             })
                             
                             return((sum(stat <= null) + 1) / (n_bootstrap + 1))
                           })
