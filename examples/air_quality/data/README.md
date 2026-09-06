# Beijing observations: provenance and reuse

Source: Chen, S. (2017), **Beijing Multi-Site Air Quality**, UCI Machine
Learning Repository, [DOI 10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G).
The [dataset page](https://archive.ics.uci.edu/dataset/501/beijing) attributes
air-quality observations to the Beijing Municipal Environmental Monitoring
Center and licenses the data under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
The data retain that license; the package's MIT license does not replace it.
No endorsement by the data creators is implied.

Changes: select calendar year 2014, four alphabetically first stations
(Aotizhongxin, Changping, Dingling, Dongsi), and PM2.5, PM10, NO2 in micrograms
per cubic metre. Average available hourly observations separately over 00–11
and 12–23 local recorded hours; retain counts, empty means, and six decimal
places. These four stations are a teaching subset, not a representative city
sample. The source has missing values; no missingness is hidden in the extract.

`provenance.json` pins both original archive and derived CSV SHA-256 hashes.
To rebuild (network needed only for this explicit preparation step):

```bash
curl -L --fail 'https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip' -o /tmp/beijing-uci501.zip
python -m examples.air_quality.prepare /tmp/beijing-uci501.zip
```

Ordinary examples use the bundled extract and require no network or account.
Do not silently accept an upstream hash change; inspect it first.
