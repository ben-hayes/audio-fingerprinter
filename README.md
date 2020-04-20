# audio-fingerprinter

Implementation of Wang's [1] constellation map fingerprinting algorithm

## Usage

A database of fingerprints can be constructed from a folder of audio documents as follows:

```python
from fingerprint_builder import fingerprintBuilder

fingerprintBuilder("/path/to/audio_files/", "/path/to/fingerprint_db.db")
```

A folder of queries can then be identified, and the top three results for each stored in a text file as follows:

```python
from audio_identification import audioIdentification

audioIdentification("/path/to/queries/", "/path/to/fingerprint_db.db", "/path/to/output.txt")
```

## References

[1] Avery  Li-Chun  Wang.  _'An  Industrial-Strength  Audio Search  Algorithm'_,  in ISMIR  2003,  4th  Symposium Conference on Music Information Retrieval, pages 7–13, 2003.
