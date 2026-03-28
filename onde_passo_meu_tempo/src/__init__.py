# Pacote src — pipeline de mobilidade "Onde Passo o Meu Tempo?"
#
# Patch de compatibilidade: trackintel 1.4.2 chama GeoDataFrame._geodataframe_constructor_with_fallback,
# um método interno que foi removido no geopandas 0.14.0. Este shim restaura o comportamento
# equivalente para que trackintel funcione com geopandas 1.x.

import pandas as pd
import geopandas as gpd

if not hasattr(gpd.GeoDataFrame, "_geodataframe_constructor_with_fallback"):

    @classmethod  # type: ignore[misc]
    def _geodataframe_constructor_with_fallback(cls, *args, **kwargs):
        try:
            return cls(*args, **kwargs)
        except Exception:
            return pd.DataFrame(*args, **kwargs)

    gpd.GeoDataFrame._geodataframe_constructor_with_fallback = (  # type: ignore[attr-defined]
        _geodataframe_constructor_with_fallback
    )
