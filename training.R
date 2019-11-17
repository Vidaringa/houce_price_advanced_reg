# Training


library(tidyverse)

# Modelling
library(tidymodels)
library(caret)
library(caretEnsemble)
# library(bartMachine)


library(doParallel)
library(parallel)

df_training <- read_csv("final_trian.csv")
df_test <- read_csv("final_test.csv")

sale_price <- df_training$sale_price
train_id <- df_training$id
test_id <- df_test$id

df_training <- df_training %>% select(-sale_price)



# Setup -------------------------------------------------------------------



ames_all <- bind_rows(df_training, df_test)

ames_all$exter_qual <- factor(ames_all$exter_qual, levels = c("Fa", "TA","Gd", "Ex"), ordered = TRUE)
ames_all$exter_cond <- factor(ames_all$exter_cond, levels = c("Po", "Fa", "TA" ,"Gd", "Ex"), ordered = TRUE)
ames_all$bsmt_qual <- factor(ames_all$bsmt_qual, levels = c("no_basement", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
ames_all$bsmt_cond <- factor(ames_all$bsmt_cond, levels = c("no_basement", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
ames_all$heating_qc <- factor(ames_all$heating_qc, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
ames_all$kitchen_qual <- factor(ames_all$kitchen_qual, levels = c("Fa", "TA", "Gd", "Ex"), ordered = TRUE)
ames_all$fireplace_qu <- factor(ames_all$fireplace_qu, levels = c("no_fireplace", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
ames_all$garage_qual <- factor(ames_all$garage_qual, levels = c("no_garage", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
ames_all$garage_cond <- factor(ames_all$garage_cond, levels = c("no_garage", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
ames_all$pool_qc <- factor(ames_all$pool_qc, levels = c("no_pool", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)

# Verð að breyta character í factor
ames_all <- ames_all %>%
        mutate_if(is.character, as.factor)

# Bý til aftur train og test set
ames_training <- ames_all %>% filter(id %in% df_training$id) %>% select(-id)
# ames_training$sale_price <- df_training$sale_price
ames_test <- ames_all %>% filter(id %in% df_test$id) %>% select(-id)



int_var <- c("bedroom_abv_gr", "bsmt_full_bath", "bsmt_half_bath", "fireplaces", "full_bath", "garage_cars",
             "half_bath", "kitchen_abv_gr", "mo_sold", "tot_rms_abv_grd", "kmeans", "ms_sub_class", "yr_sold")


to_dummy <- c("ms_zoning", "street", "alley", "lot_shape", "land_contour", "utilities", "lot_config",
              "land_slope", "neighborhood", "condition1", "condition2", "bldg_type", "house_style", "roof_style",
              "roof_matl", "exterior1st", "exterior2nd", "mas_vnr_type", "exter_qual", "exter_cond", "foundation",
              "bsmt_qual", "bsmt_cond", "bsmt_exposure", "bsmt_fin_type1", "bsmt_fin_type2", "heating", "heating_qc",
              "central_air", "electrical", "kitchen_qual", "overall_qual", "overall_cond", "functional", "fireplace_qu",
              "garage_type", "garage_finish", "garage_qual", "garage_cond", "paved_drive", "pool_qc", "fence", "misc_feature",
              "sale_type", "sale_condition")

yeo <- c("total_bsmt_sf", "x1st_flr_sf", "lot_area", "bsmt_fin_sf1", "mas_vnr_area", "bsmt_unf_sf",
         "x2nd_flr_sf", "gr_liv_area", "garage_area", "wood_deck_sf", "open_porch_sf")

to_scale <- ames_all %>% names() %>% as_tibble() %>% filter(!(value %in% to_dummy), !(value %in% int_var), value != "id")
to_scale <- to_scale$value


# Recipe ------------------------------------------------------------------

ames_recipe<- recipe(~., data = ames_training) %>%

        # impute
        step_knnimpute(all_predictors()) %>%

        # yeo johnon
        step_YeoJohnson(one_of(yeo)) %>%

        # integer values to factor
        step_num2factor(one_of(int_var)) %>%
        # ordred factor
        step_num2factor(starts_with("overall_"), ordered = TRUE) %>%

        # lump variables
        step_other(neighborhood, threshold = 0.03) %>%
        step_other(bsmt_qual, threshold = 0.05) %>%

        # dummy
        step_dummy(one_of(int_var)) %>%
        step_dummy(one_of(to_dummy)) %>%

        # Interaction
        step_interact(terms = ~ starts_with("overall_"):starts_with("neighborhood_") +
                              starts_with("overall_"):gr_liv_area +
                              starts_with("overall_"):starts_with("exter_qual_")) %>%

        step_interact(terms = ~ starts_with("neighborhood_"):gr_liv_area +
                              starts_with("neighborhood_"):starts_with("exter_qual_")) %>%
        step_interact(terms = ~ gr_liv_area:starts_with("exter_qual_")) %>%
        step_center(one_of(to_scale)) %>%
        step_scale(one_of(to_scale)) %>%
        step_knnimpute(all_predictors())

ames_recipe_nzv <- ames_recipe %>%
        step_zv(all_predictors()) %>%
        step_nzv(all_predictors())



# Prep --------------------------------------------------------------------


trained_rec <- prep(ames_recipe, training = ames_training)
trained_rec_nzv <- prep(ames_recipe_nzv, training = ames_training)

# Full data
train_data <- bake(trained_rec, new_data = ames_training)
test_data <- bake(trained_rec, new_data = ames_test)

# add id and sale_price
train_data$id <- train_id
train_data$sale_price <- sale_price
test_data$id <- test_id


# without nzv
train_data_nzv <- bake(trained_rec_nzv, new_data = ames_training)
test_data_nzv <- bake(trained_rec_nzv, new_data = ames_test)

# add sale_price
train_data_nzv$sale_price <- sale_price



# controls ----------------------------------------------------------------


ctrl <- trainControl(method = "cv",
                     number = 10,
                     allowParallel = TRUE)

ctrl_2 <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 3)

ctrl_3 <- trainControl(method = "cv",
                       number = 10,
                       allowParallel = TRUE,
                       search = "random")




# Elastic net -------------------------------------------------------------

byrja <- Sys.time()

enet_train <- train(log1p(sale_price) ~ .,
                    data = train_data_nzv,
                    method = "glmnet",
                    tuneLength = 20,
                    metric = "RMSE",
                    trControl = ctrl)

Sys.time() - byrja

saveRDS(enet_train, "glmnet.rds")
#
# enet_train <- readRDS("glmnet.rds")



# MARS --------------------------------------------------------------------

byrja <- Sys.time()

mars_train <- train(log1p(sale_price) ~ .,
                    data = train_data_nzv,
                    method = "earth",
                    tuneLength = 10,
                    metric = "RMSE",
                    trControl = ctrl)


Sys.time() - byrja

saveRDS(mars_train, "mars.rds")

# mars_train <- readRDS("mars.rds")



# KNN ---------------------------------------------------------------------

byrja <- Sys.time()

knn_train <- train(log1p(sale_price) ~ .,
                    data = train_data_nzv,
                    method = "knn",
                    tuneLength = 30,
                    metric = "RMSE",
                    trControl = ctrl_2)

Sys.time() - byrja

saveRDS(knn_train, "knn.rds")

# knn_train <- readRDS("knn.rds")




# Cubist ------------------------------------------------------------------

cub_grid <- expand.grid(committees = c(1, seq(10, 100, 10)),
                        neighbors = seq(0, 9, 1))

byrja <- Sys.time()

cubist_train <- train(log1p(sale_price) ~ .,
                    data = train_data_nzv,
                    method = "cubist",
                    tuneGrid = cub_grid,
                    metric = "RMSE",
                    trControl = ctrl)

Sys.time() - byrja


saveRDS(cubist_train, "cubist.rds")

# cubist_train <- readRDS("cubist.rds")



# SVM ---------------------------------------------------------------------

byrja <- Sys.time()

svm_train <- train(log1p(sale_price) ~ .,
                    data = train_data_nzv,
                    method = "svmRadial",
                    tuneLength = 100,
                    metric = "RMSE",
                    trControl = ctrl_3)


Sys.time() - byrja

saveRDS(svm_train, "svm_train.rds")

# svm_train <- readRDS("svm_train")


# XGBoost -----------------------------------------------------------------

byrja <- Sys.time()

xgb_train <- train(log1p(sale_price) ~ .,
                    data = train_data_nzv,
                    method = "xgbTree",
                    tuneLength = 300,
                    metric = "RMSE",
                    trControl = ctrl_3)


Sys.time() - byrja

stopCluster(cl)
# saveRDS(xgb_train, "xgb.rds")


xgb_train <- readRDS("xgb_random_3000.rds")





# Stacking ----------------------------------------------------------------

no_cores <- detectCores() - 1
cl <- makeCluster(no_cores)
registerDoParallel(cl)

my_control <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE,
                           savePredictions = "final")

x_train <- train_data_nzv %>% select(-sale_price) %>% as.data.frame()
y_train <- train_data_nzv$sale_price


model_list <- caretList(x = x_train,
                        y = log1p(y_train),
                        trControl = my_control,
                        continue_on_fail = FALSE,
                        metric = "RMSE",
                        tuneList = list(
                                glmnet = caretModelSpec(method = "glmnet", tuneGrid = data.frame(.alpha = 0.1, .lambda = 1511.753)),
                                cubist = caretModelSpec(method = "cubist", tuneGrid = data.frame(committees = 100, neighbors = 3)),
                                earth = caretModelSpec(method = "earth", tuneGrid = data.frame(nprune = 14, degree = 1)),
                                knn = caretModelSpec(method = "knn", tuneGrid = data.frame(k = 5)),
                                svm = caretModelSpec(method = "svmRadial", tuneGrid = data.frame(sigma = 0.000809491, C = 19.50694)),
                                xgb = caretModelSpec(method = "xgbTree", tuneGrid = data.frame(nrounds = 544, max_depth = 4,
                                                                                               eta = 0.06978232, gamma = 5.719467,
                                                                                               colsample_bytree = 0.6949411,
                                                                                               min_child_weight = 2, subsample = 0.7168163))
                        ))



# Stacking results --------------------------------------------------------

model_results <- data.frame(
        glmnet = min(model_list$glmnet$results$RMSE),
        cubist = min(model_list$cubist$results$RMSE),
        mars = min(model_list$mars$results$RMSE),
        knn = min(model_list$knn$results$RMSE),
        svm = min(model_list$svm$results$RMSE),
        xgb = min(model_list$xgb$results$RMSE)
)

resamples <- resamples(model_list)

dotplot(resamples, metric = "RMSE")

modelCor(resamples)




# Ensemble ----------------------------------------------------------------

ensemble_1 <- caretEnsemble(
        model_list,
        metric = "RMSE",
        trControl = my_control
)

summary(ensemble_1)

ensemble_2 <- caretStack(model_list,
                         method = "glm",
                         metric = "RMSE")

print(ensemble_2)

#
# ensemble_3 <- caretStack(model_list,
#                          method = "gbm",
#                          metric = "RMSE",
#                          verbose = FALSE,
#                          tuneLength = 10)



# Prediction --------------------------------------------------------------

final_pred <- predict(ensemble_1, newdata = test_data_nzv)


final_prediction <- tibble(ID = test_id,
                           SalePrice = as.numeric(final_pred))

write_csv(final_prediction, "final_prediction.csv")




# XGBoost - "raw" data ----------------------------------------------------

df_training <- read_csv("final_trian.csv")
df_test <- read_csv("final_test.csv")

total <- bind_rows(df_training, df_test)

total$exter_qual <- factor(total$exter_qual, levels = c("Fa", "TA","Gd", "Ex"), ordered = TRUE)
total$exter_cond <- factor(total$exter_cond, levels = c("Po", "Fa", "TA" ,"Gd", "Ex"), ordered = TRUE)
total$bsmt_qual <- factor(total$bsmt_qual, levels = c("no_basement", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
total$bsmt_cond <- factor(total$bsmt_cond, levels = c("no_basement", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
total$heating_qc <- factor(total$heating_qc, levels = c("Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
total$kitchen_qual <- factor(total$kitchen_qual, levels = c("Fa", "TA", "Gd", "Ex"), ordered = TRUE)
total$fireplace_qu <- factor(total$fireplace_qu, levels = c("no_fireplace", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
total$garage_qual <- factor(total$garage_qual, levels = c("no_garage", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
total$garage_cond <- factor(total$garage_cond, levels = c("no_garage", "Po", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)
total$pool_qc <- factor(total$pool_qc, levels = c("no_pool", "Fa", "TA", "Gd", "Ex"), ordered = TRUE)

# Verð að breyta character í factor
total <- total %>%
        mutate_if(is.character, as.factor)

df_training <- total %>% filter(id %in% df_training$id) %>% select(-id)
df_test <- total %>% filter(id %in% df_test$id) %>% select(-id)


xgb_recipe <- recipe(sale_price ~., data = df_training) %>%
        step_knnimpute(all_predictors())


xgb_rec <- prep(xgb_recipe, training = df_training)

# Full data
xgb_train_data <- bake(xgb_rec, new_data = df_training)
xgb_test_data <- bake(xgb_rec, new_data = df_test)


# add id and sale_price
xgb_train_data$id <- train_id
xgb_train_data$sale_price <- sale_price
xgb_test_data$id <- test_id



# no_cores <- detectCores() - 1
# registerDoParallel(cores=no_cores)
# cl <- makeCluster(no_cores)

byrja <- Sys.time()

xgb_train <- train(log(sale_price) ~ .,
                    data = xgb_train_data,
                    method = "xgbTree",
                    tuneLength = 10,
                    metric = "RMSE",
                    trControl = ctrl)


Sys.time() - byrja

# stopCluster(cl)
