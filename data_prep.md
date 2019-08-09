Untitled
================

## Missing data

I start by fixing the obvious missing values. There are few problematic
variables, all related to Garage.

``` r
# Missing data according to data_description.txt file

training$PoolQC[is.na(training$PoolQC)] <- "no_pool"
training$MiscFeature[is.na(training$MiscFeature)] <- "none"
training$Alley[is.na(training$Alley)] <- "no_alley"
training$Fence[is.na(training$Fence)] <- "no_fence"
training$FireplaceQu[is.na(training$FireplaceQu)] <- "no_fireplace"
training$GarageType[is.na(training$GarageType)] <- "no_garage"
training$BsmtFinType2[is.na(training$BsmtFinType2)] <- "no_basement"
training$BsmtFinType1[is.na(training$BsmtFinType1)] <- "no_basement"
training$BsmtExposure[is.na(training$BsmtExposure)] <- "no_basement"
training$BsmtQual[is.na(training$BsmtQual)] <- "no_basement"
training$BsmtCond[is.na(training$BsmtCond)] <- "no_basement"


df_missing <- map_df(training, function(x) mean(is.na(x))) %>%
        gather()

ggplot(filter(df_missing, value > 0),
       aes(x = fct_reorder(key, value),
           y = value)) +
        geom_bar(stat = "identity") +
        coord_flip()
```

![](data_prep_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

### Further investigation into missing values

``` r
convert_missing <- function(x) ifelse(is.na(x), 0, 1)

house_missing <- apply(training, 2, convert_missing)

Heatmap(house_missing,
        name = "Missing",
        column_title = "Predictors",
        row_title = "Samles",
        col = c("black", "lightgrey"),
        show_heatmap_legend = FALSE,
        row_names_gp = gpar(fontsize = 0)) # text size for row names
```

![](data_prep_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
gg_miss_upset(training)
```

![](data_prep_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->