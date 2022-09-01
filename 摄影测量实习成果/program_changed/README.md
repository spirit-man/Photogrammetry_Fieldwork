# Guidelines for this photogrammetry package
***
## Procedures are as following
### solve parameters
1. interior orientation--return interior orientation parameters
2. relative orientation--return relative orientation parameters, q, R1 and R2 
3. forward intersection--return B, Q and model coordinates
4. absolute orientation--return absolute orientation parameters and ground coordinates
### calculate ground coordinates for points on your photo
* basically the same procedures as above, using functions in corresponding classes
***
## Quick start
* load your own data in data_loading.py
* import python files to seperately excecute one of the procedures above
* remember to instantiate the class first and excecute its main function before using it
***
## Attentions
* model coordinates are measured in mm, while ground coordinates in m
* while loading excel tables, your excel should start from cell B3 by default, override or_read_excel() and write_excel() in util.py if not, and your excel should contain numbers only
* in forward intersection, all variables are measured in mm, so the limitation of Q is 1000
* uncomment code blocks to save outputs to files in main.py before excecution
***
## Unsolved problems
* do a4 in A, relative_orientation.py uses +Z1 or -Z1?
* why adding minus in front of f in L, relative_orientation.py?
##### <font color=blue>I would be grateful if you could help me with the above and modify my codes for the better!</font>
