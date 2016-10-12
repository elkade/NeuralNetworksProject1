args = commandArgs(trailingOnly = TRUE)

# test if there is at least one argument: if not, return an error
if (length(args) == 0) {
	stop("At least one argument must be supplied (input file).n", call. = FALSE)
}
path <- args[1]

MyData <- read.csv(file = path, header = TRUE, sep = ",")

#jpeg('rplot.jpg')

plot(MyData)

#dev.off()