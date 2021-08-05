import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *

df_normal = pd.DataFrame(columns = ['x' , 'L', 'F', 'U'])

#Add new ROW
df_normal.loc[0]=[ 'Normal_1', 78.9847, 78.7363, 79.3137 ]
df_normal.loc[1]=[ 'Normal_2', 78.9774, 78.8682, 79.3210 ]
df_normal.loc[2]=[ 'Normal_3', 78.9740, 78.9667, 79.3244 ]
df_normal.loc[3]=[ 'Normal_4', 78.9702, 78.8261, 79.3282 ]

df_elite = pd.DataFrame(columns = ['x' , 'L', 'F', 'U'])
#Add new ROW
df_elite.loc[0]=[ 'Elite_1', 20.0491, 20.1671, 20.3742 ]
df_elite.loc[1]=[ 'Elite_2', 20.0418, 20.3373, 20.3816 ]
df_elite.loc[2]=[ 'Elite_3', 20.0385, 20.2926, 20.3849 ]
df_elite.loc[3]=[ 'Elite_4', 20.0347, 20.4461, 20.3887 ]

fig = plt.Figure()

(ggplot(df_normal, aes(x = 'x', y = 'F'))
    + geom_point(size = 4)
    + geom_errorbar(aes(ymax = 'U', ymin = 'L'))
    + scale_color_grey() + theme_classic()
    + ylim((78, 80))
    + geom_hline(aes(yintercept = 79.1492), colour="red")
    + labs(y="Probability") + labs(x="Week")
    + labs(title="Normal TransformCard Probability"))

(ggplot(df_elite, aes(x = 'x', y = 'F'))
    + geom_point(size = 4)
    + geom_errorbar(aes(ymax = 'U', ymin = 'L'))
    + scale_color_grey() + theme_classic()
    + ylim((19.5, 20.6))
    + geom_hline(aes(yintercept = 20.2117), colour="red")
    + labs(y="Probability") + labs(x="Week")
    + labs(title="Elite TransformCard Probability"))