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

/* Left and right column */
.column.side {
  width: 25%;
}

/* Middle column */
.column.middle {
  width: 75%;
}

.card {
    border: 1px solid black;
    border-radius: 3vmax;
    position: fixed;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 90vw;
    height: 90vh;
}

.rounded {
    border: 1px solid cadetblue;
    border-radius: 30px;
}

.shaded {
    background: gainsboro;
    border: 1px solid cadetblue;
    border-radius: 30px;
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

#rcorners1 {
  border-radius: 25px 15px 20px 5px;
  background: #73AD21;
  padding: 20px;
  width: 200px;
  height: 150px;
}

#rcorners2 {
  border-radius: 25px 15px 10px;
  border: 2px solid #73AD21;
  padding: 20px;
  width: 200px;
  height: 150px;
}

#rcorners3 {
  border-radius: 25px;
  background: url(paper.gif);
  background-position: left top;
  background-repeat: repeat;
  padding: 20px;
  width: 200px;
  height: 150px;



"""
