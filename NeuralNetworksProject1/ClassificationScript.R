args = commandArgs(trailingOnly = TRUE)

# test if there is at least one argument: if not, return an error
if (length(args) == 0) {
    stop("At least one argument must be supplied (input file).n", call. = FALSE)
}
output_path <- args[1]
input_path <- args[2]

output <- read.csv(file = output_path, header = TRUE, sep = ",")
input <- read.csv(file = input_path, header = TRUE, sep = ",")
plot(output$x, output$y, pch = 21, col = c("red", "green", "blue")[unclass(output$cls)])

points(input$x, input$y, pch = 21, bg = c("red", "green", "blue")[unclass(input$cls)])
