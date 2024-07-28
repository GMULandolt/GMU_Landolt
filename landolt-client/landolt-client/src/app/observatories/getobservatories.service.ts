import { Injectable } from '@angular/core';
import { LoggerService } from '../logger.service';
import { Observatory } from './observatory';
import { Observable, catchError, map, throwError } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';

interface GetAllObservatoriesRequest { 
  max: number;
}

export interface GetAllObservatoriesResponse {
  observatories: Observatory[];
}

const httpOptions = {
  headers: new HttpHeaders({
    'Content-Type':  'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, DELETE, PUT, OPTIONS'
  })
};

@Injectable({
  providedIn: 'root'
})
export class GetobservatoriesService {

  constructor(private logger: LoggerService, private http: HttpClient) { 
  }

  getObservatories (): Observable <Observatory[]> { 
    return this.http.post<GetAllObservatoriesResponse>('/ws/getObservatories', httpOptions)
  .pipe (
    catchError((error, caught) => {
      return throwError(()=>new Error("Invalid request or server error"))
    })
  )
  .pipe<Observatory[]>(map<GetAllObservatoriesResponse, Observatory[]>(response => {
    console.log("httpresponse")
    console.log(response.observatories)
    return response.observatories;
  }))
}

}

