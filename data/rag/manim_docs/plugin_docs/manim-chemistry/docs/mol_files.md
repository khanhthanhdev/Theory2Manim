# Working with .mol files

The Manim Chemistry parsing system for mol files is still young and prone to errors. Check the examples to see what a mol file should look like. Here is the 2d morphine example:

<details>
<summary>Morphine 2D mol file</summary>

```

  Marvin  02020822302D          

 22 26  0  0  1  0            999 V2000
    0.0000   -0.8250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7145   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7145   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7145    0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7145    0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.8250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0030   -1.6464    0.0000 C   0  0  2  0  0  0  0  0  0  0  0  0
    0.7109   -2.0631    0.0000 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.7127   -2.0511    0.0000 C   0  0  1  0  0  0  0  0  0  0  0  0
    1.4268   -1.6510    0.0000 C   0  0  1  0  0  0  0  0  0  0  0  0
    0.7067   -2.8808    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5371   -1.1745    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7206   -2.8689    0.0000 C   0  0  2  0  0  0  0  0  0  0  0  0
    1.4310   -0.8297    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1419   -2.0715    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0128   -3.2891    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4365   -3.2845    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4262    0.8296    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.9416   -1.8530    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4954   -2.6258    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1397   -1.2341    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7183   -1.2330    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  7  1  1  0  0  0  0
  1  2  1  0  0  0  0
  2 12  1  0  0  0  0
  1  3  2  0  0  0  0
  3 14  1  0  0  0  0
  2  4  2  0  0  0  0
  4 18  1  0  0  0  0
  3  5  1  0  0  0  0
  4  6  1  0  0  0  0
  5  6  2  0  0  0  0
  7  8  1  0  0  0  0
  7  9  1  0  0  0  0
  7 22  1  1  0  0  0
  8 10  1  0  0  0  0
  8 11  1  0  0  0  0
  8 20  1  1  0  0  0
  9 12  1  6  0  0  0
  9 13  1  0  0  0  0
 10 15  1  1  0  0  0
 10 14  1  0  0  0  0
 11 16  2  0  0  0  0
 13 17  1  6  0  0  0
 13 16  1  0  0  0  0
 15 19  1  0  0  0  0
 21 15  1  0  0  0  0
 22 21  1  0  0  0  0
M  END
```
</details>


All mol files start with 3 lines with some info. Some of them include the software used to make it, the molecule name, etc. Others like this one just leave some useless data. 

Then, we get to the actual atoms. We have a column that starts indicating how many atoms there are in our molecule. In this case, there are 22 atoms. The next column shows how many bonds we have. In this case, there are 26 bonds.

The next thing we have are some rows with the atoms' positional data as xyz coordinates and their element. 

Finally, we have rows with all the atoms listed and their corresponding bonds. The first column of those rows indicate the "starting atom", the second one shows the atom to which they are connected to or the "ending atom". Then, we have the type of bond (single, double, triple, etc.) and then some extra data regarding aromaticity.