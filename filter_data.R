df <- t(read.table('exprMatrix.tsv', header = TRUE))
df <- df[,colMeans(df)>5]
df <- df[, rank(-apply(df, 2, var)) <= ncol(df) / 2]
write.table(df, 'transposed_expr.tsv')

