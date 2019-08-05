from schelling import *

if __name__ == '__main__':
    basic_shelling = Schelling(50, 50, 0.3, 0.2, 500, 4)
    basic_shelling.prepare()
    basic_shelling.plot('Schelling Model with 4 colors: Start State with Happiness Threshold 20%', 'basic_schelling_4_20_start.png')
    basic_shelling.update()
    basic_shelling.plot('Schelling Model with 4 colors: Final State with Happiness Threshold 20%', 'basic_schelling_4_20_final.png')

    #new_schelling = CultureSchelling(50, 50, 0.2, 0.2, 2000, [0.01, 0.02, 0.03], 3)
    #new_schelling.prepare()
    #new_schelling.plot('CultureSchelling Model with 3 different colors: Start State with Happiness Threshold 20%', 'differ_culture_schelling_3_20_start.png')
    #new_schelling.update()
    #new_schelling.plot('CultureSchelling Model with 3 different colors: Final State with Happiness Threshold 20%', 'differ_culture_schelling_3_20_final.png')


