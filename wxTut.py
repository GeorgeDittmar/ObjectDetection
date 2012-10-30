#!/usr/bin/python
import wx

class test(wx.Frame):
    def __init__(self,parent,id):
        wx.Frame.__init__(self,parent,id,"TestFrame",size=(500,500))

if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = test(parent=None,id=-1,)
    frame.show()
    app.mainloop()
