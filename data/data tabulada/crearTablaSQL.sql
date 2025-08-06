-- Table: public.results_borough_mgwr

-- DROP TABLE IF EXISTS public.results_borough_mgwr;

CREATE TABLE IF NOT EXISTS public.resultados_gwr
(
    "PRICE" double precision,
    "BROKERTITLE" text COLLATE pg_catalog."default",
    intercept double precision,
    "BEDS" double precision,
    "BATH" double precision,
    "PROPERTYSQFT" double precision,
    mi_avg double precision,
    tpa double precision,
    d_cp double precision,
    d_ts double precision,
    "CAT_VM" double precision,
    "CAT_VU" double precision,
    "CAT_Otros" double precision,
    residuals double precision,
    "BoroName" text COLLATE pg_catalog."default",
	"Latitud" double precision,
	"Longitud" double precision,
	"PRICE_2" integer
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.resultados_gwr
    OWNER to postgres;