output_path <- "classification_errors.csv"

output <- read.csv(file = output_path, header = FALSE, sep = ";")

plot(output, col = "red", cex = .2)
output_path <- "regression_errors.csv"

output <- read.csv(file = output_path, header = FALSE, sep = ";")

plot(output, col = "red", cex = .2)
