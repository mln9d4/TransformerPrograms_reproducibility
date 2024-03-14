### added by Kevin
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
### end

from programs.rasp.sort import sort
sort.run(["<s>", "3", "1", "4", "2", "4", "0", "</s>"])
