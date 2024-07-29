
export class Observatory {
   name: string;
   latitude: number;
   longitude: number; 
   timezone: string; 
   timezone_integer: number;  

   constructor (n: string, lat: number, long: number, tz: string, tzi: number) { 
        this.name = n;
        this.latitude = lat;
        this.longitude = long; 
        this.timezone = tz; 
        this.timezone_integer = tzi;  
   }
}