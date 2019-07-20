from schelling import *

if __name__ == '__main__':
    basic_shelling = Schelling(50, 50, 0.2, 0.8, 2000, 3)
    basic_shelling.prepare()
    basic_shelling.plot('Schelling Model with 3 colors: Start State with Happiness Threshold 80%', 'basic_schelling_3_80_start.png')
    basic_shelling.update()
    basic_shelling.plot('Schelling Model with 3 colors: Final State with Happiness Threshold 80%', 'basic_schelling_3_80_final.png')

    new_schelling = CultureSchelling(50, 50, 0.2, 0.8, 2000, 0.02, 3)
    new_schelling.prepare()
    new_schelling.plot('CultureSchelling Model with 3 colors: Start State with Happiness Threshold 80%', 'culture_schelling_3_80_start.png')
    new_schelling.update()
    new_schelling.plot('CultureSchelling Model with 3 colors: Final State with Happiness Threshold 80%', 'culture_schelling_3_80_final.png')


