setwd("~/Documentos/Farmacos/Dados_FBZ/Flubendazol/trim")

data_c <- read.table('cristalino')
data_m <- read.table('moinho')
data_e <- read.table('evaporacao')


plot(data_c, type = 'l', col = 'blue', ylim = c(-2,2))
par(new = T)
plot(data_m, type = 'l', col = 'green', ylim = c(-2,2))
par(new = T)
plot(data_e, type = 'l', col = 'red', ylim = c(-2,2))