## Energy Efficiency - Evaluating the impact of building design and construction
## on its heating and cooling demands and, consequently, its energy efficiency.
## Using Advanced Statistical Methods.

# Import all Libraries
library(readxl)
library(Matrix)
library(caTools)
library(splines)
library(glmnet)
library(ggplot2)
library(mgcv) # Added: Necessary for the gam() function

set.seed(123)
par(mar = c(4,4,4,4))
par(mfrow = c(1, 2))

# Get the Data
# Ensure the path to your Excel file is correct for your system
ENB2012_data <- read_excel("Desktop/Stat 444/Final Project Proposal/ENB2012_data.xlsx")

# Define predictor and response variables
myvars <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")
X <- ENB2012_data[, myvars] # Ensure X is a data frame for consistency with gam()
y_1 <- ENB2012_data$Y1
y_2 <- ENB2012_data$Y2

# This function implements K-fold CV
CV_kfold <- function(y, X_df, K) { # Renamed X to X_df to emphasize it should be a data.frame
  
  # split the data into K folds
  n <- length(y) # n data points
  idx <- sample.int(n) # random permutation of 1...n
  
  # Ensure kidx length matches n; handle cases where n is not perfectly divisible by K
  kidx_raw <- rep(1:K, each = floor(n / K))
  kidx <- c(kidx_raw, sample(1:K, n - length(kidx_raw))) # Assign remaining randomly
  kidx <- sample(kidx) # Randomize the assignments further
  
  foldidx <- split(idx, kidx)
  
  # fit the model and store prediction error
  errvec <- numeric(K)
  for (k in 1:K) {
    trainidx <- unlist(foldidx[-k]) # Use unlist to combine list elements into a single vector
    testidx <- foldidx[[k]]
    
    ytrain <- y[trainidx]
    Xtrain <- X_df[trainidx,] # Subset the data frame
    ytest <- y[testidx]
    Xtest <- X_df[testidx,] # Subset the data frame
    
    # Model fit and predict
    # Ensure formula variables match names in Xtrain/Xtest
    # Pass Xtrain as 'data' argument to gam
    model_gam <- gam(ytrain ~ s(X1, k = 10) + s(X2, k = 10) + s(X3, k = 6) + s(X4, k = 3) + X5 + s(X6, k = 3) + s(X7, k = 3) + s(X8, k = 5),
                     data = Xtrain, select = TRUE)
    # summary(model_gam) # Typically not shown inside CV loop as it's repetitive
    
    # Ensure newdata in predict is a data.frame matching the structure of X
    ytestpred <- predict(model_gam, newdata = Xtest)
    
    # Compute error (Sum of Squared Errors for each fold)
    errvec[k] <- sum((ytestpred - ytest)^2)
  }
  return(sum(errvec) / n) # Return Mean Squared Error
}

# --- Train/Test Split (for a single hold-out validation, separate from CV) ---
# If you want a train/test split for initial model fitting, use this.
# Otherwise, for pure cross-validation performance, you might not need this explicit split.

# A more robust way to do a train/test split using caTools::sample.split
# You need a response variable for sample.split to ensure stratification if desired
# Using Y1 for splitting example
split_logical <- sample.split(ENB2012_data$Y1, SplitRatio = 0.8) # Returns TRUE/FALSE vector


# --- Model Fitting on Full Data (or Train Data if using the split above) ---
# L1 Regularized - Spline Regression using mgcv software
# Fit models using the full dataset (X, y_1, y_2) for general model insights,
# or use train_data if you want to evaluate on a separate test set.
# For the CV, we pass the full X and y.

# Fitting on the full dataset for CV evaluation purposes
gam_model_y1 <- gam(y_1 ~ s(X1, k = 10) + s(X2, k = 10) + s(X3, k = 6) + s(X4, k = 3) + X5 + s(X6, k = 3) + s(X7, k = 3) + s(X8, k = 5),
                    data = X, select = TRUE) # X is already a data.frame
summary(gam_model_y1)

gam_model_y2 <- gam(y_2 ~ s(X1, k = 10) + s(X2, k = 10) + s(X3, k = 6) + s(X4, k = 3) + X5 + s(X6, k = 3) + s(X7, k = 3) + s(X8, k = 5),
                    data = X, select = TRUE) # X is already a data.frame
summary(gam_model_y2)

# --- Predictions using the fitted models ---
# Correct usage of predict: pass the model object and new data (X)
modpred_y1 <- predict(gam_model_y1, newdata = X)
modpred_y2 <- predict(gam_model_y2, newdata = X)

# --- Evaluate models using K-fold CV ---
# Pass the full X (as a data.frame) and y_1/y_2 to the CV function
print("Cross-validation MSE for Y1:")
CV_kfold(y_1, X, 10)

print("Cross-validation MSE for Y2:")
CV_kfold(y_2, X, 10)

# Plot the Graph
# For plotting we will use a train-test split

# Test Train Split
split_logical <- sample.split(ENB2012_data$Y1, SplitRatio = 0.8) # Returns TRUE/FALSE vector

train_data <- ENB2012_data[split_logical, ]
test_data <- ENB2012_data[!split_logical, ]

X_Model_train <- train_data[, myvars]
y_1_Model_train <- train_data$Y1
y_2_Model_train <- train_data$Y2

# Fitting the Model 

gam_model_y1_plot <- gam(y_1_Model_train ~ s(X1, k = 10) + s(X2, k = 10) + s(X3, k = 6) + s(X4, k = 3) + X5 + s(X6, k = 3) + s(X7, k = 3) + s(X8, k = 5),
                    data = X_Model_train, select = TRUE) # X is already a data.frame

gam_model_y2_plot <- gam(y_2_Model_train ~ s(X1, k = 10) + s(X2, k = 10) + s(X3, k = 6) + s(X4, k = 3) + X5 + s(X6, k = 3) + s(X7, k = 3) + s(X8, k = 5),
                    data = X_Model_train, select = TRUE) # X is already a data.frame

X_Model_test <- test_data[, myvars]
y_1_Model_test <- test_data$Y1
y_2_Model_test <- test_data$Y2

# Test data predictions
modpred_y1_plot <- predict(gam_model_y1_plot, newdata = X_Model_test, se.fit = TRUE)
modpred_y2_plot <- predict(gam_model_y2_plot, newdata = X_Model_test, se.fit = TRUE)


PLOTTEXTSIZE <- 2
LINEWIDTH <- 3

# Plot the Heating Effect
plot(x = y_1_Model_test,
     main = "Actual vs. Predicted Heating Load (Y1)",
     xlab = "Index",
     ylab = "Energy Required",
     pch = 19, # Use filled circles for the points
     col = "red")
lines(modpred_y1_plot$fit)


# Plot the Cooling Effect
plot(x = y_2_Model_test,
     main = "Actual vs. Predicted Cooling Load (Y2)",
     xlab = "Index",
     ylab = "Energy Required",
     pch = 19, # Use filled circles for the points
     col = "blue")
lines(modpred_y2_plot$fit)







