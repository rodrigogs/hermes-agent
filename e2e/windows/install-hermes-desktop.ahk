#Requires AutoHotkey v2.0
#SingleInstance Force

; enable click animation ;;;;;;;;;;;


DllCall("SystemParametersInfo", UInt, 0x101D, UInt, 0, UInt, 1, UInt, 0) ;SPI_SETMOUSESONAR ON

OnExit(ExitSub)
ExitSub:
DllCall("SystemParametersInfo", UInt, 0x101D, UInt, 0, UInt, 0, UInt, 0) ;SPI_SETMOUSESONAR OFF
ExitApp


~LButton::
Send {Ctrl}
return

~LButton UP::
Send {Ctrl}
return
;;;;;;;;;;;


; Wait for the Hermes installer window to appear.
winTitle := "Hermes"
if not WinWait(winTitle,, 30)
{
    FileAppend("ERROR: Hermes installer window did not appear within 30s`n", "ahk.log")
    ExitApp(1)
}

Sleep(1000)

WinGetPos(&x, &y, &w, &h, winTitle)
FileAppend(Format("Window found at x={1} y={2} w={3} h={4}`n", x, y, w, h), "ahk.log")

; click install
clickX := x + (w / 2)
clickY := y + 418
Click(clickX, clickY)

; done
ExitApp(0)