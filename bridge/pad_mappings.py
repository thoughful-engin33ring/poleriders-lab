"""Controller mappings for the game's 2014-era SDL, which predates the DualSense: without one SDL sees the pad as a
plain joystick and the game ignores it. (Through Steam, Steam Input hides this by presenting a virtual pad; our
launches start the game binary directly.) Passed as SDL_GAMECONTROLLERCONFIG by prmod.py run and
spr_env.launch_game. GUIDs are this SDL's format (vendor, product little-endian); the layout is the DS4 one the game
ships (macOS IOKit enumerates DualSense buttons/axes in the same order). Check with the bridge's `joysticks`."""

_PS = ("a:b1,b:b2,back:b8,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b4,leftstick:b10,"
       "lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b11,righttrigger:a4,rightx:a2,righty:a5,"
       "start:b9,x:b0,y:b3,")
PAD_MAPPINGS = "\n".join([
    "4c05000000000000e60c000000000000,DualSense Wireless Controller," + _PS,
    "4c05000000000000f20d000000000000,DualSense Edge Wireless Controller," + _PS,
    "4c05000000000000cc09000000000000,PS4 Controller (v2)," + _PS,
])
