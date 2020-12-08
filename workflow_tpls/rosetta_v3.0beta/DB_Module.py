'''
    Rosetta version 3-alpha (3a) 
    Pedotransfer functions by Schaap et al., 2001 and Zhang and Schaap, 2016.
    Copyright (C) 2016  Marcel G. Schaap

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    Marcel G. Schaap can be contacted at:
    mschaap@cals.arizona.edu

'''



import os
import sys
#note that sqlite3 and mysqldb get imported in DB_init CONDITIONALLY

class DB(object):

    @property
    def conn(self): return(self._conn)
    @conn.setter
    def conn(self,value): self._conn=value

    def __init__(self,host, user, db_name, sqlite_path=None, debug=False):

        self.debug=debug

        if sqlite_path=='':  # there is probably a more elegant way, but --sqlite was defined with a default of ''
            path=None
        else:
            path=sqlite_path

        if path is not None:

            if not os.path.exists(path):
                print(("Cannot find the sqlite path %s" % (path)))
                sys.exit(1)

            try:
                import sqlite3
            except ImportError:
                print("Python sqlite3 module not installed?")
                sys.exit (1)
                
            self.Error = sqlite3.Error

            try:
                self.conn = sqlite3.connect(sqlite_path)
            except self.Error as e:
                print("Database connection error %d: %s" % (e.args[0], e.args[1]))
                sys.exit (1)

            #self.conn.text_factory = str # needed to get the binary data from sqlite
            self.conn.text_factory = bytes
            #self.conn.text_factory = lambda x: str(x, 'iso-8859-1')
            self.sqlite=True
        else:

            try:
                import MySQLdb
            except ImportError:
                print("Python MySQLdb module not installed?")
                sys.exit (1)

            try:
                import getpass
            except ImportError:
                print("Python getpass module not installed?")
                sys.exit (1)

            self.Error = MySQLdb.Error

            print("Password for Rosetta database on MySQL server:")
            passwd = getpass.getpass()
            try:
                self.conn=MySQLdb.connect(host = host,
                                          user = user,
                                          passwd = passwd,
                                          db = db_name)
                #self.conn.text_factory = str 
                

            except self.Error as e:
                print("Database connection error %d: %s" % (e.args[0], e.args[1]))
                sys.exit (1)

            self.conn.text_factory = bytes
            self.sqlite=False
            
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.commit()
        self.close()
        if self.debug:
            print("Closed DB object")

    def get_cursor(self):
        try:
            cursor = self.conn.cursor ()
        except self.Error as e:
            print("Database cursor error %d: %s" % (e.args[0], e.args[1]))
            sys.exit (1)
        return(cursor)

    def commit(self): 
        if self.conn: self.conn.commit()

    def close(self):  
        if self.conn: self.conn.close()
