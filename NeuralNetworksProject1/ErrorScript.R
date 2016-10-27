args = commandArgs(trailingOnly = TRUE)

# test if there is at least one argument: if not, return an error
if (length(args) == 0) {
	stop("At least one argument must be supplied (input file).n", call. = FALSE)
}
output_path <- args[1]

output <- read.csv(file = output_path, header = FALSE, sep = ",")

plot(input, type = "p", col = "black", cex = .2)