style="""* {
  box-sizing: border-box;
}

body {
  margin: 0;
}

/* Style the header */
.header {
  /*background-color: #f1f1f1;*/
  background-color: #b6b5ba;
  padding: 20px;
  text-align: center;
}

/* Style the top navigation bar */
.topnav {
  overflow: hidden;
  background-color: #333;
}

/* Style the topnav links */
.topnav a {
  float: left;
  display: block;
  color: #f2f2f2;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;
}

/* Change color on hover */
.topnav a:hover {
  background-color: #ddd;
  color: black;
}

/* Create three unequal columns that floats next to each other */
.column {
  float: left;
  padding: 10px;
}
.column.left {
  background-color: aliceblue;
  width: 25%;
  height: 500px;
  overflow: auto;
}
.column.middle {
  border: 1px solid black;
  width: 70%;
  background-color: cornslik;
  height: 500px;
  overflow: auto;
  border-radius: 30px;
}
.column.small {
  width: 5%;
}

.column.middle_top {

  border: 0px;
  height: 40%;
  width: 100%;
  overflow: auto;
}

.column.middle_middle {
  height: 50%;
  width: 100%;
  border-radius: 10px;
  background-color: blanchedalmond;
  overflow: auto;
}

.column.middle_bottom{
  border: 0px solid orange;
  height: 10%;
  overflow: auto;
  width: 100%;
}


.column.middle_big {
  width: 90%;
  border: 1px solid orange;
  border-radius 10px;
  padding: 10px;
}



.rounded {
    border: 1px solid cadetblue;
    height:380px;
    border-radius: 15px;
}

.shaded {
    background: gainsboro;
    border: 1px solid cadetblue;
    border-radius: 10px;
}

img {
    border: 1px solid #e3e3e3;
    /*border: 1px solid #b6b5ba;*/
}


/* Clear floats after the columns */
.row::after {
  content: "";
  display: table;
  clear: both;
}





"""
