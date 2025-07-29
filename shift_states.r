

rot <- function(a) matrix(c(cos(a), sin(a), -sin(a), cos(a)), 2, 2)


shift_states <- function(shapefile) {
    alaska <- shapefile |> filter(STATEFP == "02")
    alaska_g <- st_geometry(alaska)
    alaska_centroid <- st_centroid(st_union(alaska_g))
    alaska_trans <- (alaska_g - alaska_centroid) *
                    rot(-39 * pi / 180) / 2.3 +
                    alaska_centroid + c(1000000, -5100000)
    alaska <- alaska %>%
        st_set_geometry(alaska_trans) %>%
        st_set_crs(st_crs(shapefile))


    hawaii <- shapefile %>% filter(STATEFP == "15")
    hawaii_g <- st_geometry(hawaii)
    hawaii_centroid <- st_centroid(st_union(hawaii_g))
    hawaii_trans <- (hawaii_g - hawaii_centroid) *
        rot(-35 * pi / 180) + hawaii_centroid + c(5200000, -1400000)
    hawaii <- hawaii %>%
        st_set_geometry(hawaii_trans) %>%
        st_set_crs(st_crs(shapefile))


    puerto <- shapefile %>% filter(STATEFP == "72")
    puerto_g <- st_geometry(puerto)
    puerto_centroid <- st_centroid(st_union(puerto_g))
    puerto_trans <- puerto_g + c(-1100000, 500000.0)
    puerto <- puerto %>%
        st_set_geometry(puerto_trans) %>%
        st_set_crs(st_crs(shapefile))


    df <- shapefile[!shapefile$STATEFP %in% c("60", "66", "69", "72", "79", "78", "15", "02"), ]


    df_final <- rbind(df, alaska, hawaii, puerto)
}