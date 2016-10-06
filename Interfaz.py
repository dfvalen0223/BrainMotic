#!/usr/bin/env python     #solution
# -*- coding: utf-8 -*-

import kivy
from kivy.app import App

#para que no aparezca la pantalla completa

# !/usr/bin/python # -*- coding: utf-8 -*- from kivy.config import Config Config.set('graphics', 'fullscreen', '0') from kivy.app import App class TestApp(App): pass if __name__ == '__main__': TestApp().run()


from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, WipeTransition
from kivy.properties import NumericProperty,StringProperty
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.config import Config



class ZeroScreen(Screen):
    pass


class FirstScreen(Screen):
    pass

class Light_1(Screen):

    Light1_In=0
    if (Light1_In==1):
        State="ON_L1"
    elif(Light1_In==0):
        State="OFF_L1"

class Light1_On(Screen):
    pass

class Light1_Off(Screen):
    pass


class  Socket_1(Screen):
    Socket1_In = 1
    if (Socket1_In == 1):
        State_S1 = "ON_S1"
    elif (Socket1_In == 0):
        State_S1 = "OFF_S1"

class Socket1_On(Screen):
    pass

class Socket1_Off(Screen):
    pass


class Lux_1(Screen):
    pass

class Lux1_value(Screen):

    TXT_L=StringProperty()
    L_imput = 200

    def __init__(self, *args, **kwargs):
        super(Lux1_value, self).__init__(*args, **kwargs)
        self.TXT_L=str(self.L_imput)

    if L_imput < 100:  # down temperature
        R = .0
        G = .0
        B = .0
    elif L_imput >= 100 and L_imput < 150:
        R = .9
        G = .7
        B = .0
    elif L_imput >= 150 and L_imput <= 200:
        R = .9
        G = .2
        B = .0
    elif L_imput > 200:
        R = .6
        G = .0
        B = .0

class  Temperature_1(Screen):
    pass

class  Temp1_value(Screen):

    TXT_T = StringProperty()
    T_imput = 100

    def __init__(self, *args, **kwargs):
        super(Temp1_value, self).__init__(*args, **kwargs)
        self.TXT_T = str(self.T_imput)

    if T_imput < 17:  # down temperature
        RT = .1
        GT = .8
        BT = .9
    elif T_imput >= 17 and T_imput < 24:
        RT = .0
        GT = .5
        BT = 1
    elif T_imput >= 24 and T_imput <= 30:
        RT = .5
        GT = .3
        BT = .8
    elif T_imput > 30:
        RT = .9
        GT = .0
        BT = .0

class Back_1(Screen):
    pass

class SecondScreen(Screen):
    # -------------
    # --LUX-----------------
    TXT_L = StringProperty()

    for i in range(45,150,2):
        L_imput = i
    # ------------TEMPERATURE--------------

    TXT_T = StringProperty()
    T_imput =12

    def __init__(self, *args, **kwargs):
        super(SecondScreen, self).__init__(*args, **kwargs)
        self.TXT_L = str(self.L_imput)
        self.TXT_T = str(self.T_imput)

    if L_imput < 100:  # down temperature
        R = .0
        G = .0
        B = .0
    elif L_imput >= 100 and L_imput < 150:
        R = .9
        G = .7
        B = .0
    elif L_imput >= 150 and L_imput <= 200:
        R = .9
        G = .2
        B = .0
    elif L_imput > 200:
        R = .6
        G = .0
        B = .0
# ------------TEMPERATURE--------------
    if T_imput < 17:  # down temperature
        RT = .1
        GT = .8
        BT = .9
    elif T_imput >= 17 and T_imput < 24:
        RT = .0
        GT = .5
        BT = 1
    elif T_imput >= 24 and T_imput <= 30:
        RT = .5
        GT = .3
        BT = .8
    elif T_imput > 30:
        RT = .9
        GT = .0
        BT = .0

class Light_2(Screen):

    Light2_In=1
    if (Light2_In==1):
        State_L2="ON_L2"
    elif(Light2_In==0):
        State_L2="OFF_L2"


class Light2_On(Screen):
    pass

class Light2_Off(Screen):
    pass

class  Socket_2(Screen):

    Socket2_In = 1
    if (Socket2_In == 1):
        State_S2 = "ON_S2"
    elif (Socket1_In == 0):
        State_S2 = "OFF_S2"

class  Socket2_On(Screen):
    pass

class  Socket2_Off(Screen):
    pass

class  Lux_2(Screen):
    pass

class  Lux2_value(Screen):
    TXT_L2 = StringProperty()
    L2_imput = 200

    def __init__(self, *args, **kwargs):
        super(Lux2_value, self).__init__(*args, **kwargs)
        self.TXT_L2 = str(self.L2_imput)

    if L2_imput < 100:  # down temperature
        R = .0
        G = .0
        B = .0
    elif L2_imput >= 100 and L2_imput < 150:
        R = .9
        G = .7
        B = .0
    elif L2_imput >= 150 and L2_imput <= 200:
        R = .9
        G = .2
        B = .0
    elif L2_imput > 200:
        R = .6
        G = .0
        B = .0

    pass

class  Temperature_2(Screen):
    pass

class Temp2_value(Screen):
    TXT_T2 = StringProperty()
    T2_imput = 300

    def __init__(self, *args, **kwargs):
        super(Temp2_value, self).__init__(*args, **kwargs)
        self.TXT_T2 = str(self.T2_imput)

    if T2_imput < 17:  # down temperature
        RT = .1
        GT = .8
        BT = .9
    elif T2_imput >= 17 and T2_imput < 24:
        RT = .0
        GT = .5
        BT = 1
    elif T2_imput >= 24 and T2_imput <= 30:
        RT = .5
        GT = .3
        BT = .8
    elif T2_imput > 30:
        RT = .9
        GT = .0
        BT = .0

class Back_2(Screen):
    pass

class ThirdScreen(Screen):
    pass

class Light_3(Screen):

    Light3_In=0

    if (Light3_In==1):
        State_L3="ON_L3"
    elif(Light3_In==0):
        State_L3="OFF_L3"

class Light3_On(Screen):
    pass



class Light3_Off(Screen):
    pass

class  Socket_3(Screen):
    Socket3_In = 1
    if (Socket3_In == 1):
        State_S3 = "ON_S3"
    elif (Socket3_In == 0):
        State_S3 = "OFF_S3"

class Socket3_On(Screen):
    pass

class Socket3_Off(Screen):
    pass


class Lux_3(Screen):
    pass

class Lux3_value(Screen):

    TXT_L3=StringProperty()
    L3_imput = 200

    def __init__(self, *args, **kwargs):
        super(Lux3_value, self).__init__(*args, **kwargs)
        self.TXT_L3=str(self.L3_imput)

    if L3_imput < 100:  # down temperature
        R = .0
        G = .0
        B = .0
    elif L3_imput >= 100 and L3_imput < 150:
        R = .9
        G = .7
        B = .0
    elif L3_imput >= 150 and L3_imput <= 200:
        R = .9
        G = .2
        B = .0
    elif L3_imput > 200:
        R = .6
        G = .0
        B = .0

class  Temperature_3(Screen):
    pass

class  Temp3_value(Screen):

    TXT_T3 = StringProperty()
    T3_imput = 100

    def __init__(self, *args, **kwargs):
        super(Temp3_value, self).__init__(*args, **kwargs)
        self.TXT_T3 = str(self.T3_imput)

    if T3_imput < 17:  # down temperature
        RT = .1
        GT = .8
        BT = .9
    elif T3_imput >= 17 and T3_imput < 24:
        RT = .0
        GT = .5
        BT = 1
    elif T3_imput >= 24 and T3_imput <= 30:
        RT = .5
        GT = .3
        BT = .8
    elif T3_imput > 30:
        RT = .9
        GT = .0
        BT = .0

class Back_3(Screen):
    pass

class MyScreenManager(ScreenManager):
    pass

Interfaz=Builder.load_file("brainmotic1.kv")


# OTHER WAY TO PUT THE CLASS THAT CONTEIN THE KIVY LENGUAJE
#class MyWidget(BoxLayout):
#    pass

class BrainMoticApp(App):

    def build(self):
        return Interfaz


    def on_pause(self):
        return True

if __name__ == "__main__":
    BrainMoticApp().run()




